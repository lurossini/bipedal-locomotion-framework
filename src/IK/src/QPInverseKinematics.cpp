/**
 * @file QPInverseKinematics.cpp
 * @authors Giulio Romualdi
 * @copyright 2021 Istituto Italiano di Tecnologia (IIT). This software may be modified and
 * distributed under the terms of the BSD-3-Clause license.
 */

#include <cstddef>
#include <map>
#include <memory>

#include <iDynTree/KinDynComputations.h>

#include <OsqpEigen/OsqpEigen.h>

#include <BipedalLocomotion/IK/IKLinearTask.h>
#include <BipedalLocomotion/IK/IntegrationBasedIK.h>
#include <BipedalLocomotion/IK/QPInverseKinematics.h>
#include <BipedalLocomotion/System/ConstantWeightProvider.h>
#include <BipedalLocomotion/System/VariablesHandler.h>
#include <BipedalLocomotion/System/WeightProvider.h>
#include <BipedalLocomotion/TextLogging/Logger.h>

using namespace BipedalLocomotion::IK;
using namespace BipedalLocomotion;

struct QPInverseKinematics::Impl
{
    struct TaskWithPriority
    {
        std::shared_ptr<QPInverseKinematics::Task> task;
        std::size_t priority;
        std::shared_ptr<const System::WeightProviderPort> weightProvider;
        Eigen::MatrixXd tmp; /**< This is a temporary matrix usefull to reduce dynamics allocation
                                in advance method */
        Eigen::VectorXd achievedOutput;
    };

    QPInverseKinematics::State solution;

    System::VariablesHandler::VariableDescription robotVelocityVariable;

    bool isVerbose{false};

    std::unordered_map<std::string, TaskWithPriority> tasks;

    std::vector<std::reference_wrapper<const TaskWithPriority>> constraints;
    std::vector<std::reference_wrapper<TaskWithPriority>> costs;
    bool isValid{false};
    bool isInitialized{false};
    bool isFinalized{false};

    struct PriorityLevel
    {
        OsqpEigen::Solver solver;
        std::vector<std::reference_wrapper<const TaskWithPriority>> constraints;
        std::vector<std::reference_wrapper<TaskWithPriority>> costs;
        std::size_t numberOfConstraints{0};

        Eigen::MatrixXd hessian;
        Eigen::VectorXd gradient;
        Eigen::MatrixXd constraintMatrix;
        Eigen::VectorXd lowerBound;
        Eigen::VectorXd upperBound;

        bool isFirstIteration{true};

        bool initializeSolver()
        {
            constexpr auto logPrefix = "[QPInversekinematics::Impl::initializeSolver]";
            // Hessian matrix
            Eigen::SparseMatrix<double> hessianSparse = this->hessian.sparseView();
            if (!this->solver.data()->setHessianMatrix(hessianSparse))
            {
                log()->error("{} Unable to set the hessian matrix.", logPrefix);
                return false;
            }

            // gradient
            if (!this->solver.data()->setGradient(gradient))
            {
                log()->error("{} Unable to set the gradient vector.", logPrefix);
                return false;
            }

            // if there are constraints
            if (this->numberOfConstraints > 0)
            {
                Eigen::SparseMatrix<double> constraintsMatrixSparse
                    = this->constraintMatrix.sparseView();
                if (!this->solver.data()->setLinearConstraintsMatrix(constraintsMatrixSparse))
                {
                    log()->error("{} Unable to set the constraint matrix.", logPrefix);
                    return false;
                }

                if (!this->solver.data()->setBounds(this->lowerBound, this->upperBound))
                {
                    log()->error("{} Unable to set the bounds.", logPrefix);
                    return false;
                }
            }

            if (!this->solver.initSolver())
            {
                log()->error("{} Unable to initialize the solver.", logPrefix);
                return false;
            }

            return true;
        }

        bool updateSolver()
        {
            constexpr auto logPrefix = "[QPInversekinematics::Impl::updateSolver]";
            // Hessian matrix
            Eigen::SparseMatrix<double> hessianSparse = this->hessian.sparseView();
            if (!this->solver.updateHessianMatrix(hessianSparse))
            {
                log()->error("{} Unable to set the hessian matrix.", logPrefix);
                return false;
            }

            // gradient
            if (!this->solver.updateGradient(gradient))
            {
                log()->error("{} Unable to set the gradient vector.", logPrefix);
                return false;
            }

            // if the number of constraints is equal to zero we do not need to update the constraint
            if (this->numberOfConstraints == 0)
            {
                return true;
            }

            // in this case the number of constraint is different from zero, so we have to update the
            // constraint matrix and the bounds
            Eigen::SparseMatrix<double> constraintsMatrixSparse = this->constraintMatrix.sparseView();
            if (!this->solver.updateLinearConstraintsMatrix(constraintsMatrixSparse))
            {
                log()->error("{} Unable to set the constraint matrix.", logPrefix);
                return false;
            }

            if (!this->solver.updateBounds(this->lowerBound, this->upperBound))
            {
                log()->error("{} Unable to set the bounds.", logPrefix);
                return false;
            }

            return true;
        }

    };
    std::vector<std::unique_ptr<PriorityLevel>> solvers;
};

QPInverseKinematics::QPInverseKinematics()
{
    m_pimpl = std::make_unique<QPInverseKinematics::Impl>();
}

QPInverseKinematics::~QPInverseKinematics() = default;

bool QPInverseKinematics::addTask(std::shared_ptr<QPInverseKinematics::Task> task,
                                  const std::string& taskName,
                                  std::size_t priority,
                                  Eigen::Ref<const Eigen::VectorXd> weight)
{
    return this->addTask(task,
                         taskName,
                         priority,
                         std::make_shared<System::ConstantWeightProvider>(weight));
}

bool QPInverseKinematics::addTask(std::shared_ptr<QPInverseKinematics::Task> task,
                                  const std::string& taskName,
                                  std::size_t priority,
                                  std::shared_ptr<const System::WeightProviderPort> weightProvider)
{
    constexpr auto logPrefix = "[QPInverseKinematics::addTask]";

    // check if the task already exist
    const bool taskExist = (m_pimpl->tasks.find(taskName) != m_pimpl->tasks.end());
    if (taskExist)
    {
        log()->error("{} The task named {} already exist.", logPrefix, taskName);
        return false;
    }

    if (priority > 0 && task->type() == System::LinearTask::Type::inequality)
    {
        log()->error("{} - [Task name: '{}'] This implementation of the inverse kinematics cannot "
                     "handle inequality tasks with positive priority.",
                     logPrefix,
                     taskName);
        return false;
    }

    // Store the task inside the InverseKinematics
    m_pimpl->tasks[taskName].task = task;
    m_pimpl->tasks[taskName].priority = priority;

    if (priority > 0 && !weightProvider)
    {
        log()->error("{} - [Task name: '{}'] Please provide the associated weight. This is "
                     "necessary since the priority of the task is positive",
                     logPrefix,
                     taskName);

        // erase the task since it is not valid
        m_pimpl->tasks.erase(taskName);

        return false;
    }

    if (priority > 0 && weightProvider)
    {
        if (weightProvider->getOutput().size() != task->size())
        {
            log()->error("{} - [Task name: '{}'] The size of the weight is not coherent with the "
                         "size of the task. Expected: {}. Given: {}.",
                         logPrefix,
                         taskName,
                         m_pimpl->tasks[taskName].task->size(),
                         weightProvider->getOutput().size());

            // erase the task since it is not valid
            m_pimpl->tasks.erase(taskName);

            return false;
        }

        // add the weight In this case we assume that the weight will be constant
        m_pimpl->tasks[taskName].weightProvider = weightProvider;

        // add the task to the list of the element that are used to build the cost function
        m_pimpl->costs.push_back(m_pimpl->tasks[taskName]);
    } else
    {
        m_pimpl->constraints.push_back(m_pimpl->tasks[taskName]);
    }

    // if you add a new task you should call finalize again
    m_pimpl->isFinalized = false;

    return true;
}

bool QPInverseKinematics::setTaskWeight(
    const std::string& taskName, std::shared_ptr<const System::WeightProviderPort> weightProvider)
{
    constexpr auto logPrefix = "[QPInverseKinematics::setTaskWeight]";

    auto tmp = m_pimpl->tasks.find(taskName);

    const bool taskExist = (tmp != m_pimpl->tasks.end());
    if (!taskExist)
    {
        log()->error("{} The task named {} does not exist.", logPrefix, taskName);
        return false;
    }

    auto taskWithPriority = tmp->second;

    if (taskWithPriority.priority == 0)
    {
        log()->error("{} - [Task name: '{}'] The weight can be set only to a task with priority "
                     "greater than 0.",
                     logPrefix,
                     taskName);
        return false;
    }

    if (weightProvider == nullptr || !weightProvider->isOutputValid())
    {
        log()->error("{} - [Task name: '{}'] The weightProvider is not valid.",
                     logPrefix,
                     taskName);
        return false;
    }

    if (weightProvider->getOutput().size() != taskWithPriority.task->size())
    {
        log()->error("{} - [Task name: '{}'] The size of the weight is not coherent with the "
                     "size of the task. Expected: {}. Given: {}.",
                     logPrefix,
                     taskName,
                     taskWithPriority.task->size(),
                     weightProvider->getOutput().size());
        return false;
    }

    // update the weight
    m_pimpl->tasks[taskName].weightProvider = weightProvider;

    return true;
}

bool QPInverseKinematics::setTaskWeight(const std::string& taskName,
                                        Eigen::Ref<const Eigen::VectorXd> weight)
{
    return this->setTaskWeight(taskName, std::make_shared<System::ConstantWeightProvider>(weight));
}

std::weak_ptr<const System::WeightProviderPort>
QPInverseKinematics::getTaskWeightProvider(const std::string& taskName) const
{
    constexpr auto logPrefix = "[QPInverseKinematics::getTaskWeightProvider]";

    auto taskWithPriority = m_pimpl->tasks.find(taskName);
    const bool taskExist = (taskWithPriority != m_pimpl->tasks.end());
    if (!taskExist)
    {
        log()->error("{} The task named {} does not exist.", logPrefix, taskName);
        return std::shared_ptr<System::WeightProviderPort>();
    }

    return taskWithPriority->second.weightProvider;
}

bool QPInverseKinematics::finalize(const System::VariablesHandler& handler)
{
    constexpr auto logPrefix = "[QPInverseKinematics::finalize]";

    if (!m_pimpl->isInitialized)
    {
        log()->error("{} Please call initialize() before finalize().", logPrefix);
        return false;
    }

    m_pimpl->isFinalized = false;

    m_pimpl->isValid = false;
    m_pimpl->solvers.clear();

    // set the variable handler for all the tasks
    for (auto& [name, task] : m_pimpl->tasks)
    {
        if (!task.task->setVariablesHandler(handler))
        {
            log()->error("{} Error while setting the variable handler in the task named {}, having "
                         "the following description {}.",
                         logPrefix,
                         name,
                         task.task->getDescription());
            return false;
        }

    }

    // resize the temporary matrix usefull to reduce dynamics allocation when advance() is called
    for (auto& cost : m_pimpl->costs)
    {
        if (cost.get().weightProvider == nullptr)
        {
            log()->error("{} One of the weight provider has been not correctly set.", logPrefix);
            return false;
        }

        cost.get().tmp.resize(handler.getNumberOfVariables(),
                              cost.get().weightProvider->getOutput().size());
        cost.get().achievedOutput.resize(cost.get().task->size());
    }

    std::map<std::size_t, std::vector<std::reference_wrapper<Impl::TaskWithPriority>>> levels;
    for (auto& cost : m_pimpl->costs)
    {
        levels[cost.get().priority].push_back(cost);
    }
    if (levels.empty())
    {
        levels[1];
    }

    auto constraints = m_pimpl->constraints;
    for (const auto& [priority, costs] : levels)
    {
        auto level = std::make_unique<Impl::PriorityLevel>();
        level->costs = costs;
        level->constraints = constraints;
        for (const auto& constraint : constraints)
        {
            level->numberOfConstraints += constraint.get().task->size();
        }
        level->solver.settings()->setVerbosity(m_pimpl->isVerbose);
        level->solver.data()->setNumberOfVariables(handler.getNumberOfVariables());
        level->solver.data()->setNumberOfConstraints(level->numberOfConstraints);
        level->hessian.resize(handler.getNumberOfVariables(), handler.getNumberOfVariables());
        level->gradient.resize(handler.getNumberOfVariables());
        level->constraintMatrix.resize(level->numberOfConstraints, handler.getNumberOfVariables());
        level->upperBound.resize(level->numberOfConstraints);
        level->lowerBound.resize(level->numberOfConstraints);
        m_pimpl->solvers.push_back(std::move(level));
        for (const auto& cost : costs)
        {
            constraints.push_back(cost.get());
        }
    }

    if (!handler.getVariable(m_pimpl->robotVelocityVariable.name, m_pimpl->robotVelocityVariable))
    {
        log()->error("{} Error while retrieving the robot velocity variable.", logPrefix);
        return false;
    }

    m_pimpl->isFinalized = true;

    return true;
}

std::vector<std::string> QPInverseKinematics::getTaskNames() const
{
    std::vector<std::string> tasksName;

    for (const auto& [name, task] : m_pimpl->tasks)
    {
        tasksName.push_back(name);
    }

    return tasksName;
}

bool QPInverseKinematics::advance()
{
    constexpr auto logPrefix = "[QPInverseKinematics::advance]";

    // when advance is called the previous solution is no more valid
    m_pimpl->isValid = false;

    if (!m_pimpl->isInitialized)
    {
        log()->error("{} Please call initialize() before advance().", logPrefix);
        return false;
    }

    if (!m_pimpl->isFinalized)
    {
        log()->error("{} Please call finalize() before advance().", logPrefix);
        return false;
    }

    // update of all the tasks
    for (auto& [name, task] : m_pimpl->tasks)
    {
        if (!task.task->update())
        {
            log()->error("{} Unable to update the task named {}.", logPrefix, name);
            return false;
        }

        // the outcome of isValid() should be the same of update. This test is required
        assert(task.task->isValid() && "One of the task is not valid.");
    }

    for (auto& levelPtr : m_pimpl->solvers)
    {
        auto& level = *levelPtr;
        // Compute the gradient and the hessian
        level.hessian.setZero();
        level.gradient.setZero();
        for (auto& cost : level.costs)
        {
            Eigen::Ref<const Eigen::MatrixXd> A = cost.get().task->getA();
            Eigen::Ref<const Eigen::VectorXd> b = cost.get().task->getB();

            // Here we avoid to have dynamic allocation
            cost.get().tmp.noalias() = A.transpose() * //
                                       cost.get().weightProvider->getOutput().asDiagonal();
            level.hessian.noalias() += cost.get().tmp * A;
            level.gradient.noalias() -= cost.get().tmp * b;
        }

        // compute the constraints
        std::size_t index = 0;
        for (const auto& constraint : level.constraints)
        {
            // check if the number of constraints changed
            if (level.numberOfConstraints < index + constraint.get().task->size())
            {
                log()->error("{} The number of constraints changed. Please call finalize() again. "
                             "Expected number of constraints: {}. Adding the constrained named {}, the "
                             "number of constraints becomes {}.",
                             logPrefix,
                             level.numberOfConstraints,
                             constraint.get().task->getDescription(),
                             index + constraint.get().task->size());
                return false;
            }

            Eigen::Ref<const Eigen::MatrixXd> A = constraint.get().task->getA();
            Eigen::Ref<const Eigen::VectorXd> b = constraint.get().task->getB();

            level.constraintMatrix.middleRows(index, constraint.get().task->size()) = A;
            level.upperBound.segment(index, constraint.get().task->size()) = b;

            if (constraint.get().priority > 0)
            {
                level.upperBound.segment(index, constraint.get().task->size())
                    = constraint.get().achievedOutput;
                const auto& weights = constraint.get().weightProvider->getOutput();
                for (Eigen::Index row = 0; row < weights.size(); ++row)
                {
                    if (weights[row] == 0)
                    {
                        level.constraintMatrix.row(index + row).setZero();
                        level.upperBound[index + row] = 0;
                    }
                }
            }

            if (constraint.get().task->type() == System::LinearTask::Type::inequality)
            {
                level.lowerBound.segment(index, constraint.get().task->size())
                    .setConstant(-OsqpEigen::INFTY);
            } else
            {
                assert(constraint.get().task->type() == System::LinearTask::Type::equality);

                level.lowerBound.segment(index, constraint.get().task->size())
                    = level.upperBound.segment(index, constraint.get().task->size());
            }

            index += constraint.get().task->size();
        }

        if (index != level.numberOfConstraints)
        {
            log()->error("{} The number of constraints changed. Please call finalize() again. "
                         "Expected number of constraints: {}. Computed number of constraints: {}.",
                         logPrefix,
                         level.numberOfConstraints,
                         index);
            return false;
        }

        // update the solver
        if (!level.isFirstIteration)
        {
            if (!level.updateSolver())
            {
                log()->error("{} Unable to update the QP solver.", logPrefix);
                return false;
            }
        } else
        {
            if (!level.initializeSolver())
            {
                log()->error("{} Unable to run the QP solver the first time.", logPrefix);
                return false;
            }
            level.isFirstIteration = false;
        }

        // solve the QP
        if (level.solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError)
        {
            log()->error("{} Unable to to solve the problem.", logPrefix);
            return false;
        }

        if (level.solver.getStatus() != OsqpEigen::Status::Solved
            && level.solver.getStatus() != OsqpEigen::Status::SolvedInaccurate)
        {
            log()->error("{} osqp was not able to find a feasible solution.", logPrefix);
            return false;
        }

        if (level.solver.getStatus() == OsqpEigen::Status::SolvedInaccurate)
        {
            log()->debug("{} The solver found an inaccurate feasible solution.", logPrefix);
        }

        for (auto& cost : level.costs)
        {
            cost.get().achievedOutput.noalias() = cost.get().task->getA() * level.solver.getSolution();
        }
    }

    // retrieve the solution
    constexpr std::size_t spatialVelocitySize = 6;
    const std::size_t joints = m_pimpl->robotVelocityVariable.size - spatialVelocitySize;

    m_pimpl->solution.jointVelocity
        = m_pimpl->solvers.back()->solver.getSolution().segment(m_pimpl->robotVelocityVariable.offset
                                                    + spatialVelocitySize,
                                                joints);

    m_pimpl->solution.baseVelocity.coeffs()
        = m_pimpl->solvers.back()->solver.getSolution().segment<spatialVelocitySize>(
            m_pimpl->robotVelocityVariable.offset);

    m_pimpl->isValid = true;

    return true;
}

bool QPInverseKinematics::isOutputValid() const
{
    return m_pimpl->isValid;
}

const QPInverseKinematics::State& QPInverseKinematics::getOutput() const
{
    return m_pimpl->solution;
}

std::weak_ptr<QPInverseKinematics::Task> QPInverseKinematics::getTask(const std::string& name) const
{
    constexpr auto logPrefix = "[QPInverseKinematics::getTask]";
    auto task = m_pimpl->tasks.find(name);

    if (task == m_pimpl->tasks.end())
    {
        log()->warn("{} The task named {} does not exist. A nullptr will be returned.",
                    logPrefix,
                    name);
        return std::shared_ptr<QPInverseKinematics::Task>(nullptr);
    }

    return task->second.task;
}

bool QPInverseKinematics::initialize(
    std::weak_ptr<const ParametersHandler::IParametersHandler> handler)
{
    constexpr auto logPrefix = "[QPInverseKinematics::initialize]";

    auto ptr = handler.lock();
    if (ptr == nullptr)
    {
        log()->error("{} The parameter handler is not valid.", logPrefix);
        return false;
    }

    if (!ptr->getParameter("robot_velocity_variable_name", m_pimpl->robotVelocityVariable.name))
    {
        log()->error("{} Error while retrieving the robot velocity variable.", logPrefix);
        return false;
    }

    if (!ptr->getParameter("verbosity", m_pimpl->isVerbose))
    {
        log()->info("{} 'verbosity' not found. The following parameter will be used '{}'.",
                    logPrefix,
                    m_pimpl->isVerbose);
    }

    m_pimpl->isInitialized = true;

    return true;
}

std::string QPInverseKinematics::toString() const
{
    std::ostringstream oss;

    oss << "====== QPInverseKinematics class ======" << std::endl
        << "The optimization problem is composed by the following tasks:" << std::endl;
    for (const auto& [name, task] : m_pimpl->tasks)
    {
        oss << "\t - " << name << ": " << task.task->getDescription()
            << " Priority: " << task.priority << "." << std::endl;
    }
    oss << "Note: The lower is the integer associated to the priority, the higher is the priority."
        << std::endl
        << "==========================" << std::endl;

    return oss.str();
}

Eigen::Ref<const Eigen::VectorXd> QPInverseKinematics::getRawSolution() const
{
    if (m_pimpl->solvers.empty())
    {
        static const Eigen::VectorXd empty;
        return empty;
    }
    return m_pimpl->solvers.back()->solver.getSolution();
}

IntegrationBasedIKProblem
QPInverseKinematics::build(std::weak_ptr<const ParametersHandler::IParametersHandler> handler,
                           std::shared_ptr<iDynTree::KinDynComputations> kinDyn)
{
    constexpr auto logPrefix = "[QPInverseKinematics::build]";
    auto ptr = handler.lock();

    if (ptr == nullptr)
    {
        log()->error("{} Invalid parameter handler.", logPrefix);
        return IntegrationBasedIKProblem();
    }

    if (kinDyn == nullptr || !kinDyn->isValid())
    {
        log()->error("{} Invalid iDynTree::KinDynComputations object.", logPrefix);
        return IntegrationBasedIKProblem();
    }

    std::unique_ptr<QPInverseKinematics> solver = std::make_unique<QPInverseKinematics>();
    std::unordered_map<std::string, std::shared_ptr<System::WeightProvider>> weights;

    if (!solver->initialize(ptr->getGroup("IK")))
    {
        log()->error("{} Unable to initialize the IK solver.", logPrefix);
        return IntegrationBasedIKProblem();
    }

    BipedalLocomotion::System::VariablesHandler variablesHandler;
    if (!variablesHandler.addVariable(solver->m_pimpl->robotVelocityVariable.name,
                                      kinDyn->getNrOfDegreesOfFreedom() + 6))
    {
        log()->error("{} Unable to add the variable named '{}'.",
                     logPrefix,
                     solver->m_pimpl->robotVelocityVariable.name);
        return IntegrationBasedIKProblem();
    }

    std::vector<std::string> tasks;
    if (!ptr->getParameter("tasks", tasks))
    {
        log()->error("{} Unable to find the parameter 'tasks'.", logPrefix);
        return IntegrationBasedIKProblem();
    }

    for (const auto& taskGroupName : tasks)
    {
        auto taskGroupTmp = ptr->getGroup(taskGroupName).lock();
        if (taskGroupTmp == nullptr)
        {
            log()->error("{} Unable to find the group named '{}'.", logPrefix, taskGroupName);
            return IntegrationBasedIKProblem();
        }

        // add robot_velocity_variable_name parameter since it is required by all the tasks
        auto taskGroup = taskGroupTmp->clone();
        taskGroup->setParameter("robot_velocity_variable_name",
                                solver->m_pimpl->robotVelocityVariable.name);

        // create an instance of the task
        std::string taskType;
        if (!taskGroup->getParameter("type", taskType))
        {
            log()->error("{} Unable to find the parameter 'type' in the group named '{}'.",
                         logPrefix,
                         taskGroupName);
            return IntegrationBasedIKProblem();
        }

        std::shared_ptr<IKLinearTask> taskInstance = IKLinearTaskFactory::createInstance(taskType);
        if (taskInstance == nullptr)
        {
            log()->error("{} The task type '{}' has not been registered.", logPrefix, taskType);
            return IntegrationBasedIKProblem();
        }

        if (!taskInstance->setKinDyn(kinDyn))
        {
            log()->error("{} Unable to set the kinDynComputations object for the task in the group "
                         "'{}'.",
                         logPrefix,
                         taskGroupName);
            return IntegrationBasedIKProblem();
        }

        if (!taskInstance->initialize(taskGroup))
        {
            log()->error("{} Unable to task named '{}'.", logPrefix, taskGroupName);
            return IntegrationBasedIKProblem();
        }

        int priority{0};
        if (!taskGroup->getParameter("priority", priority))
        {
            log()->error("{} Unable to get the parameter 'priority' for the task in the group "
                         "'{}'.",
                         logPrefix,
                         taskGroupName);
            return IntegrationBasedIKProblem();
        }

        if (priority < 0)
        {
            log()->error("{} Invalid priority provided for the task named '{}'. "
                         "The priority must be nonnegative.",
                         logPrefix,
                         taskGroupName);
            return IntegrationBasedIKProblem();
        }

        if (priority > 0)
        {
            std::string weightProviderType = "ConstantWeightProvider";
            if (!taskGroup->getParameter("weight_provider_type", weightProviderType))
            {
                log()->warn("{} Unable to get the parameter 'weight_provider_type' for the task "
                            "in the group "
                            "'{}'. The default one will be used. Default: '{}'.",
                            logPrefix,
                            taskGroupName,
                            weightProviderType);
            }

            auto weightProvider = BipedalLocomotion::System::WeightProviderFactory::createInstance(
                weightProviderType);

            if (weightProvider == nullptr)
            {
                log()->error("{} The weight provider '{}' has not been registered.",
                             logPrefix,
                             weightProviderType);
                return IntegrationBasedIKProblem();
            }

            if (!weightProvider->initialize(taskGroup))
            {
                log()->error("{} Unable to initialize the weight provider for the task in the "
                             "group "
                             "'{}'.",
                             logPrefix,
                             taskGroupName);
                return IntegrationBasedIKProblem();
            }

            if (!solver->addTask(taskInstance, taskGroupName, priority, weightProvider))
            {
                log()->error("{} Unable to add the task named '{}' in the solver.",
                             logPrefix,
                             taskGroupName);
                return IntegrationBasedIKProblem();
            }

            weights[taskGroupName] = weightProvider;

        } else
        {
            if (!solver->addTask(taskInstance, taskGroupName, priority))
            {
                log()->error("{} Unable to add the task named '{}' in the solver.",
                             logPrefix,
                             taskGroupName);
                return IntegrationBasedIKProblem();
            }
        }
    }

    if (!solver->finalize(variablesHandler))
    {
        log()->error("{} Unable to finalize the solver.", logPrefix);
        return IntegrationBasedIKProblem();
    }

    IntegrationBasedIKProblem problem;
    problem.ik = std::move(solver);
    problem.variablesHandler = std::move(variablesHandler);
    problem.weights = std::move(weights);

    return problem;
}
